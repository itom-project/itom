    //leaffilterproxymodel.h
    #ifndef LEAFFILTERPROXYMODEL_H
    #define LEAFFILTERPROXYMODEL_H
     
    #include <QSortFilterProxyModel>
     
    class LeafFilterProxyModel : public QSortFilterProxyModel
    {
    Q_OBJECT
    public:
        explicit LeafFilterProxyModel(QObject *parent = 0);
     
    signals:
     
    public slots:
     
    protected:
        bool filterAcceptsRow(int source_row, const QModelIndex &source_parent) const;
        bool filterAcceptsRowItself(int source_row, const QModelIndex &source_parent) const;
        bool hasAcceptedChildren(int source_row, const QModelIndex &source_parent) const;
     
    };
    #endif // LEAFFILTERPROXYMODEL_H
